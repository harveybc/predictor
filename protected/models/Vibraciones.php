<?php

class Vibraciones extends CActiveRecord {

    public static function model($className=__CLASS__) {
        return parent::model($className);
    }

    public function tableName() {
        return 'vibraciones';
    }

    public function rules() {
        return array(
            array('OT', 'numerical', 'integerOnly' => true),
            array('TAG', 'length', 'max' => 50),
            array('VibLL', 'length', 'max' => 19),
            array('VibLA, Temperatura', 'length', 'max' => 18),
                         array('Estado', 'numerical', 'integerOnly'=>true),
                        array('Observaciones', 'length', 'max'=>250),
           array('Fecha', 'safe'),
            array(' OT, Estado, Observaciones,id, Toma, TAG, Fecha, OT, VibLL, VibLA, Temperatura', 'safe', 'on' => 'search'),
        );
    }

    public function relations() {
        return array(
            'relacVibraciones' => array(self::BELONGS_TO, '', '', 'on' => ''),
        );
    }

    public function behaviors() {
        return array('CAdvancedArBehavior',
            array('class' => 'ext.CAdvancedArBehavior')
        );
    }

    public function attributeLabels() {
        return array(
            'id' => Yii::t('app', 'ID'),
            'Toma' => Yii::t('app', 'Toma'),
            'TAG' => Yii::t('app', 'Motor'),
            'Fecha' => Yii::t('app', 'Fecha'),
            'OT' => Yii::t('app', 'OT'),
            'VibLL' => Yii::t('app', 'Vibr. Motor'),
            'VibLA' => Yii::t('app', 'Vibr. Lado  Bomba o Reductor'),
            'Temperatura' => Yii::t('app', 'Temperatura'),
            'Estado' => Yii::t('app', 'Estado'),
            'Observaciones' => Yii::t('app', 'Observaciones'),            
        );
    }

    public function search() {
        $criteria = new CDbCriteria;



        //$criteria->compare('Toma',$this->Toma);

        $criteria->compare('TAG', $this->TAG, true);

        $criteria->compare('Fecha', $this->Fecha, true);

        $criteria->compare('OT', $this->OT);

        $criteria->compare('Estado', $this->Estado);

        /* 	
          $criteria->compare('id',$this->id,true);
          $criteria->compare('VibLL',$this->VibLL,true);

          $criteria->compare('VibLA',$this->VibLA,true);

          $criteria->compare('Temperatura',$this->Temperatura,true);
         */
        return new CActiveDataProvider(get_class($this), array(
                    'criteria' => $criteria,
                ));
    }

    public function searchFechas($TAG) {
        if ($TAG == "")
            $TAG = "ZZ_ZZ_Z";
        //$criteria = new CDbCriteria;
        //$criteria->compare('TAG', $TAG, true);
        
        return new CActiveDataProvider(get_class($this), array(
                    'criteria' => array('order'=>'Fecha DESC','condition'=>'TAG="'.$TAG.'"'),
                    'pagination' => array(
                        'pageSize' => 20
                    )
                ));
        /*
          $sql = 'SELECT * FROM motores WHERE Equipo="' . $equipo . '" AND Area="'.$area.'" ORDER BY Equipo';
          $myDataProvider = new CSqlDataProvider($sql, array (
          'keyField' => 'id',
          'pagination' => array (
          'pageSize' => 10,
          ),
          )
          );
          return $myDataProvider;
         * 
         */
    }

}

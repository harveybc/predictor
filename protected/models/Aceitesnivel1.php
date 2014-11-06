<?php

class Aceitesnivel1 extends CActiveRecord {

    public static function model($className=__CLASS__) {
        return parent::model($className);
    }

    public function tableName() {
        return 'aceitesnivel1';
    }

    public function rules() {
        return array(
            array('Toma, OT', 'numerical', 'integerOnly' => true),
            array('TAG, Tipo, Analista', 'length', 'max' => 50),
            array('Medicion', 'length', 'max' => 18),
            array('Estado', 'numerical', 'integerOnly'=>true),
            array('Observaciones', 'length', 'max'=>250),
            array('Fecha', 'safe'),
            array('id, Toma, TAG, Fecha, OT, Estado, Observaciones, Medicion, Tipo, Analista', 'safe', 'on' => 'search'),
        );
    }

    public function relations() {
        return array(
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
            'TAG' => Yii::t('app', 'Tag'),
            'Fecha' => Yii::t('app', 'Fecha'),
            'OT' => Yii::t('app', 'Orden de Trabajo'),
            'Estado' => Yii::t('app', 'Estado de Aceite'),
            'Observaciones' => Yii::t('app', 'Observaciones'),            
            'Medicion' => Yii::t('app', 'Valor Medicion'),
            'Tipo' => Yii::t('app', 'Tipo de Aceite'),
            'Analista' => Yii::t('app', 'Analista'),
        );
    }

    public function search() {
        $criteria = new CDbCriteria;

        $criteria->compare('id', $this->id, true);

        //$criteria->compare('Toma',$this->Toma);
        $criteria->compare('Analista', $this->Analista, true);

        $criteria->compare('TAG', $this->TAG, true);
        $criteria->compare('Fecha', $this->Fecha, true);

        $criteria->compare('OT', $this->OT);

        $criteria->compare('Estado', $this->Estado);
        /*


          $criteria->compare('Estado',$this->Estado);

          $criteria->compare('Medicion',$this->Medicion,true);

          $criteria->compare('Tipo',$this->Tipo,true);


         */
        return new CActiveDataProvider(get_class($this), array(
                    'criteria' => $criteria,
                ));
    }
    
    public function searchFechas($TAG) {
        if ($TAG == "")
            $TAG = "ZZ_ZZ_Z";
        //$criteria = new CDbCriteria;
        //$criteria->compare('TAG', $TAG, false);
        return new CActiveDataProvider(get_class($this), array(
                    //'criteria' => $criteria,
                    'criteria' => array('order'=>'Fecha DESC','condition'=>'TAG="'.$TAG.'"'),
                ));

        $sql = 'SELECT * FROM motores WHERE Equipo="' . $equipo . '" AND Area="' . $area . '" ORDER BY Equipo';
        $myDataProvider = new CSqlDataProvider($sql, array(
                    'keyField' => 'id',
                    'pagination' => array(
                        'pageSize' => 10,
                    ),
                        )
        );
        return $myDataProvider;
    }

}

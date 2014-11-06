<?php

class Motores extends CActiveRecord {

    public static function model($className=__CLASS__) {
        return parent::model($className);
    }

    public function tableName() {
        return 'motores';
    }
    //public $plan_mant_vibraciones;
    public function rules() {
        return array(
            array('TAG', 'required'),
            array('Codigo,kW, Velocidad', 'numerical'),
            array('TAG, Proceso, Area, Equipo, Marca, Modelo, Serie, Rod_LC, Rod_LA, Lubricante, IP, Frame', 'length', 'max' => 50),
            array('plan_mant_vibraciones, plan_mant_aislamiento,plan_mant_lubricantes,plan_mant_termografia', 'numerical', 'integerOnly'=>true),
            array('Motor, PathFoto', 'length', 'max' => 255),
            array('plan_mant_vibraciones,Codigo, id, TAG, Proceso, Area, Equipo, Motor, kW, Velocidad, Marca, Modelo, Serie, Rod_LC, Rod_LA, Lubricante, IP, Frame, PathFoto', 'safe', 'on' => 'search'),
        );
    }

    public function relations() {
        return array(
                //'relacMotores' => array(self::BELONGS_TO, 'Motores', '','on'=>'TAG=TAG'),
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
            'TAG' => Yii::t('app', 'TAG'),
            'Proceso' => Yii::t('app', 'Area'),
            'Area' => Yii::t('app', 'Proceso'),
            'Equipo' => Yii::t('app', 'Equipo'),
            'Motor' => Yii::t('app', 'Motor(Nombre)'),
            'kW' => Yii::t('app', 'K W'),
            'Velocidad' => Yii::t('app', 'Velocidad'),
            'Marca' => Yii::t('app', 'Marca'),
            'Modelo' => Yii::t('app', 'Modelo'),
            'Serie' => Yii::t('app', 'Serie'),
            'Rod_LC' => Yii::t('app', 'Rod Lc'),
            'Rod_LA' => Yii::t('app', 'Rod La'),
            'Lubricante' => Yii::t('app', 'Lubricante'),
            'IP' => Yii::t('app', 'Ip'),
            'Frame' => Yii::t('app', 'Frame'),
            'PathFoto' => Yii::t('app', 'Path Foto'),
            'Codigo' => Yii::t('app', 'Código'),
            'plan_mant_vibraciones' => Yii::t('app', 'Plan de mantenimiento de Vibraciones'),
            'plan_mant_aislamiento' => Yii::t('app', 'Plan de mantenimiento de Aislamiento'),
            'plan_mant_lubricantes' => Yii::t('app', 'Plan de mantenimiento de Lubricantes'),
            'plan_mant_termografia' => Yii::t('app', 'Plan de mantemimiento de Termografía'),
        );
    }

    public function search() {
        $criteria = new CDbCriteria;

        $criteria->compare('Marca', $this->Marca, true);

        $criteria->compare('Modelo', $this->Modelo, true);

        $criteria->compare('Serie', $this->Serie, true);

        $criteria->compare('Lubricante', $this->Lubricante, true);

        $criteria->compare('PathFoto', $this->PathFoto, true);


        $criteria->compare('Motor', $this->Motor, true);

        $criteria->compare('TAG', $this->TAG, true);
        
        $criteria->compare('Codigo', $this->Codigo, true);
        /*
          $criteria->compare('kW',$this->kW);

          $criteria->compare('Velocidad',$this->Velocidad);


          $criteria->compare('Rod_LC',$this->Rod_LC,true);

          $criteria->compare('Rod_LA',$this->Rod_LA,true);



          $criteria->compare('IP',$this->IP,true);

          $criteria->compare('Frame',$this->Frame,true);


         * 
         *  $criteria->compare('id',
          $this->id,
          true);

          //

          $criteria->compare('Proceso',
          $this->Proceso,
          true);

          $criteria->compare('Area',
          $this->Area,
          true);

          $criteria->compare('Equipo',
          $this->Equipo,
          true);
         */
        return new CActiveDataProvider(get_class($this), array(
                    'criteria' => $criteria,
                ));
    }

    public function searchEquipos($area, $equipo) {
        $criteria = new CDbCriteria;
        $criteria->compare('Area', $area, true);
        $criteria->compare('Equipo', $equipo, true);
        return new CActiveDataProvider(get_class($this), array(
                    'criteria' => $criteria,
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

    public function searchEquiposArea($area) {
        $criteria = new CDbCriteria;
        $criteria->compare('Area', $area, true);
        return new CActiveDataProvider(get_class($this), array(
                    'criteria' => $criteria,
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


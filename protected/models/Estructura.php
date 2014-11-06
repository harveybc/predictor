<?php

class Estructura extends CActiveRecord
{
	public static function model($className=__CLASS__)
	{
		return parent::model($className);
	}

	public function tableName()
	{
		return 'estructura';
	}

	public function rules()
	{
		return array(
                        array('Area, Proceso,Equipo', 'required'),
			array('plan_mant_ultrasonido,Codigo, Indicativo', 'numerical', 'integerOnly'=>true),
			array('Proceso, Area, Equipo', 'length', 'max'=>50),
			array('plan_mant_ultrasonido, id, Proceso, Area, Codigo, Equipo, Indicativo', 'safe', 'on'=>'search'),
		);
	}

	public function relations()
	{
		return array(
		);
	}

	public function behaviors()
	{
		return array('CAdvancedArBehavior',
				array('class' => 'ext.CAdvancedArBehavior')
				);
	}

	public function attributeLabels()
	{
		return array(
			'id' => Yii::t('app', 'ID'),
			'Proceso' => Yii::t('app', 'Área'),
			'Area' => Yii::t('app', 'Proceso'),
			'Codigo' => Yii::t('app', 'Código SAP'),
			'Equipo' => Yii::t('app', 'Equipo'),
			'Indicativo' => Yii::t('app', 'Indicativo'),
                        'plan_mant_ultrasonido' => Yii::t('app', 'Plan de mantenimiento de Ultrasonido'),
		);
	}

	public function search()
	{
		$criteria=new CDbCriteria;

		$criteria->compare('id',$this->id,true);

		$criteria->compare('Proceso',$this->Proceso,true);

		$criteria->compare('Area',$this->Area,true);

		$criteria->compare('Codigo',$this->Codigo);

		$criteria->compare('Equipo',$this->Equipo,true);

	//	$criteria->compare('Indicativo',$this->Indicativo);

		return new CActiveDataProvider(get_class($this), array(
			'criteria'=>$criteria,
		));
	}
        
         public function searchEstructura($Codigo)
         { if ($Codigo == "")
            $Codigo = "ZZ_ZZ_Z";
        $criteria = new CDbCriteria;
        $criteria->compare('Area', $Codigo);
        return new CActiveDataProvider('Estructura', array(
                    'criteria' => $criteria,
                    //'pagination' => array(
                    //    'pageSize' => 20
                   // )
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

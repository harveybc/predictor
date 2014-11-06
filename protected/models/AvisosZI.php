<?php

class AvisosZI extends CActiveRecord
{
	public static function model($className=__CLASS__)
	{
		return parent::model($className);
	}

	public function tableName()
	{
		return 'avisosZI';
	}

	public function rules()
	{
		return array(
			array('Ruta,Codigo, Operador, Estado', 'required'),
			array('Estado,Codigo, arreglado, plan_mant, OT', 'numerical', 'integerOnly'=>true),
			array('Ruta', 'length', 'max'=>128),
			array('Operador', 'length', 'max'=>64),
			array('Observaciones', 'length', 'max'=>255),
			array('id,Codigo,Ruta, Operador, Fecha, Estado, Observaciones, arreglado, plan_mant, OT', 'safe', 'on'=>'search'),
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
			'Ruta' => Yii::t('app', 'Medición'),
			'Operador' => Yii::t('app', 'Operador'),
			'Fecha' => Yii::t('app', 'Fecha'),
			'Estado' => Yii::t('app', 'Estado'),
			'Observaciones' => Yii::t('app', 'Observaciones'),
			'arreglado' => Yii::t('app', 'Arreglado'),
			'plan_mant' => Yii::t('app', 'Plan de Mantenimiento'),
			'OT' => Yii::t('app', 'Orden de Trabajo'),
                        'Código' => Yii::t('app', 'Código SAP'),
		);
	}

	public function search()
	{
		$criteria=new CDbCriteria;

		//$criteria->compare('id',$this->id,true);

		$criteria->compare('Ruta',$this->Ruta,true);

		//$criteria->compare('Operador',$this->Operador,true);

		$criteria->compare('Fecha',$this->Fecha,true);

		$criteria->compare('Estado',$this->Estado);

		//$criteria->compare('Observaciones',$this->Observaciones,true);

		//$criteria->compare('arreglado',$this->arreglado);

		//$criteria->compare('plan_mant',$this->plan_mant);
                
                //$criteria->compare('Codigo',$this->plan_mant);

		$criteria->compare('OT',$this->OT);

		return new CActiveDataProvider(get_class($this), array(
			'criteria'=>$criteria,
		));
	}
}

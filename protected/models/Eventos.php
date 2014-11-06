<?php

class Eventos extends CActiveRecord
{
	public static function model($className=__CLASS__)
	{
		return parent::model($className);
	}

	public function tableName()
	{
		return 'eventos';
	}

	public function rules()
	{
		return array(
			array('usuario', 'length', 'max'=>31),
			array('modulo, operacion', 'length', 'max'=>32),
			array('ip', 'length', 'max'=>12),
			array('descripcion', 'length', 'max'=>255),
			array('fecha', 'safe'),
			array('id, usuario, modulo, operacion, ip, descripcion, fecha', 'safe', 'on'=>'search'),
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
			'usuario' => Yii::t('app', 'Usuario'),
			'modulo' => Yii::t('app', 'Módulo'),
			'operacion' => Yii::t('app', 'Operación'),
			'ip' => Yii::t('app', 'Dirección IP'),
			'descripcion' => Yii::t('app', 'Descripción'),
			'fecha' => Yii::t('app', 'Fecha'),
		);
	}

	public function search()
	{
		$criteria=new CDbCriteria;

		$criteria->compare('id',$this->id,true);

		$criteria->compare('usuario',$this->usuario,true);

		$criteria->compare('modulo',$this->modulo,true);

		$criteria->compare('operacion',$this->operacion,true);

		$criteria->compare('ip',$this->ip,true);

		$criteria->compare('descripcion',$this->descripcion,true);

		$criteria->compare('fecha',$this->fecha,true);
                $criteria->order='Fecha Desc';

		return new CActiveDataProvider(get_class($this), array(
                  
			'criteria'=>$criteria,
		));
	}
}
